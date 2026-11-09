# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned native likelihood corpora with lm-eval v0.4.12 prompt semantics."""

from dataclasses import dataclass, replace
from functools import partial
from typing import Literal

from marin.evaluation.records import PromptFormat
from taskcompendium.evaluation.corpus import chat_variant_item
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
from taskcompendium.importers.evalchemy.math import gsm8k_cot_template
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
from experiments.post_training.task_curation.datasets.evalchemy.protocols import choice_source, fewshot_generation_source
from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.evaluation import (
    EvalDataset,
    EvaluationBenchmark,
    ProtocolVariant,
    curate,
    fixed_benchmarks,
)
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
GSM8K_TEST_FILE = "main/test-00000-of-00001.parquet"
GSM8K_TRAIN_FILE = "main/train-00000-of-00001.parquet"
GSM8K_SOURCE = HfSource("openai/gsm8k", GSM8K_REVISION, (GSM8K_TEST_FILE,), SourceFormat.PARQUET)


@dataclass(frozen=True)
class ChoiceDeclaration:
    name: str
    benchmark: str
    source: HfSource
    population: int
    shots: tuple[int, ...]
    pool_file: str
    normalizations: tuple[LikelihoodNormalization, ...]
    metric_kind: MetricKind = MetricKind.BINARY
    scoring: Literal["most_likely", "probability_mass"] = "most_likely"
    fewshot_version: str = "2026.11.02"


@dataclass(frozen=True)
class GenerationDeclaration:
    name: str
    source: HfSource
    population: int
    shots: tuple[int, ...]
    pool_file: str
    grader: Environment
    metrics: tuple[str, ...]
    primary_metric: str
    stop: tuple[str, ...]
    chat_options: dict[int, dict[str, bool | str]]


CHOICE_PROTOCOLS = (
    ChoiceDeclaration(
        benchmark="arc-challenge",
        name="arc_challenge",
        source=HfSource(
            "allenai/ai2_arc",
            "210d026faf9955653af8916fad021475a3f00453",
            ("ARC-Challenge/test-00000-of-00001.parquet",),
            SourceFormat.PARQUET,
        ),
        population=1172,
        shots=(25, 10),
        pool_file="ARC-Challenge/train-00000-of-00001.parquet",
        normalizations=(
            LikelihoodNormalization.RAW,
            LikelihoodNormalization.CHARACTERS,
        ),
    ),
    ChoiceDeclaration(
        benchmark="arc-easy",
        name="arc_easy",
        source=HfSource(
            "allenai/ai2_arc",
            "210d026faf9955653af8916fad021475a3f00453",
            ("ARC-Easy/test-00000-of-00001.parquet",),
            SourceFormat.PARQUET,
        ),
        population=2376,
        shots=(0, 10),
        pool_file="ARC-Easy/train-00000-of-00001.parquet",
        normalizations=(
            LikelihoodNormalization.RAW,
            LikelihoodNormalization.CHARACTERS,
        ),
    ),
    ChoiceDeclaration(
        benchmark="hellaswag",
        name="hellaswag",
        source=HfSource(
            "Rowan/hellaswag",
            "218ec52e09a7e7462a5400043bb9a69a41d06b76",
            ("data/validation-00000-of-00001.parquet",),
            SourceFormat.PARQUET,
        ),
        population=10042,
        shots=(10, 0),
        pool_file="data/train-00000-of-00001.parquet",
        normalizations=(
            LikelihoodNormalization.RAW,
            LikelihoodNormalization.CHARACTERS,
        ),
    ),
    ChoiceDeclaration(
        benchmark="boolq",
        name="boolq",
        source=HfSource(
            "aps/super_glue",
            "3de24cf8022e94f4ee4b9d55a6f539891524d646",
            ("boolq/validation-00000-of-00001.parquet",),
            SourceFormat.PARQUET,
        ),
        population=3270,
        shots=(0, 10),
        pool_file="boolq/train-00000-of-00001.parquet",
        normalizations=(LikelihoodNormalization.RAW,),
        fewshot_version="2026.11.05.2",
    ),
    ChoiceDeclaration(
        benchmark="piqa",
        name="piqa",
        source=HfSource(
            "baber/piqa", "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1", ("piqa_validation.parquet",), SourceFormat.PARQUET
        ),
        population=1838,
        shots=(0, 10),
        pool_file="piqa_train.parquet",
        normalizations=(
            LikelihoodNormalization.CHARACTERS,
            LikelihoodNormalization.RAW,
        ),
        fewshot_version="2026.11.05.2",
    ),
    ChoiceDeclaration(
        benchmark="openbookqa",
        name="openbookqa",
        source=HfSource(
            "allenai/openbookqa",
            "388097ea7776314e93a529163e0fea805b8a6454",
            ("main/test-00000-of-00001.parquet",),
            SourceFormat.PARQUET,
        ),
        population=500,
        shots=(0,),
        pool_file="",
        normalizations=(
            LikelihoodNormalization.RAW,
            LikelihoodNormalization.CHARACTERS,
        ),
    ),
    ChoiceDeclaration(
        benchmark="winogrande",
        name="winogrande",
        source=HfSource(
            "allenai/winogrande",
            "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
            ("winogrande_xl/validation-00000-of-00001.parquet",),
            SourceFormat.PARQUET,
        ),
        population=1267,
        shots=(5, 0),
        pool_file="winogrande_xl/train-00000-of-00001.parquet",
        normalizations=(LikelihoodNormalization.RAW,),
    ),
    ChoiceDeclaration(
        benchmark="truthfulqa",
        name="truthfulqa_mc2",
        source=HfSource(
            "truthfulqa/truthful_qa",
            "741b8276f2d1982aa3d5b832d3ee81ed3b896490",
            ("multiple_choice/validation-00000-of-00001.parquet",),
            SourceFormat.PARQUET,
        ),
        population=817,
        shots=(0, 6),
        pool_file="",
        normalizations=(LikelihoodNormalization.RAW,),
        metric_kind=MetricKind.CONTINUOUS,
        scoring="probability_mass",
        fewshot_version="2026.11.05.2",
    ),
)

GENERATION_PROTOCOLS = (
    GenerationDeclaration(
        name="gsm8k",
        source=GSM8K_SOURCE,
        population=1319,
        shots=(5, 0),
        pool_file="main/train-00000-of-00001.parquet",
        grader=MATH_GRADER,
        metrics=("exact_match,strict-match", "exact_match,flexible-extract"),
        primary_metric="exact_match,strict-match",
        stop=GSM8K_REQUEST_STOPS,
        chat_options={0: {"enable_thinking": False}, 5: {}},
    ),
    GenerationDeclaration(
        name="triviaqa",
        source=HfSource(
            "mandarjoshi/trivia_qa",
            "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
            ("rc.nocontext/validation-00000-of-00001.parquet",),
            SourceFormat.PARQUET,
        ),
        population=17944,
        shots=(5, 0),
        pool_file="rc.nocontext/train-00000-of-00001.parquet",
        grader=ALIAS_GRADER,
        metrics=("exact_match,strict_answer", "exact_match,extract_answer"),
        primary_metric="exact_match,strict_answer",
        stop=COMPLETION_REQUEST_STOPS,
        chat_options={5: {"enable_thinking": False}},
    ),
    GenerationDeclaration(
        name="nq_open",
        source=HfSource(
            "google-research-datasets/nq_open",
            "5dd9790a83002ad084ddeb7c420dc716852c6f28",
            ("nq_open/validation-00000-of-00001.parquet",),
            SourceFormat.PARQUET,
        ),
        population=3610,
        shots=(5, 0),
        pool_file="nq_open/train-00000-of-00001.parquet",
        grader=ALIAS_GRADER,
        metrics=("exact_match,strict_answer", "exact_match,extract_answer"),
        primary_metric="exact_match,strict_answer",
        stop=COMPLETION_REQUEST_STOPS,
        chat_options={5: {}},
    ),
    GenerationDeclaration(
        name="drop",
        source=HfSource(
            "EleutherAI/drop",
            "619386fc7ffe6dace3cc6d3483c69b164ebf6bf3",
            ("drop_validation.parquet",),
            SourceFormat.PARQUET,
        ),
        population=9536,
        shots=(3, 0),
        pool_file="drop_train.parquet",
        grader=BASIC_GRADER,
        metrics=("em,extract_answer", "f1,extract_answer"),
        primary_metric="f1,extract_answer",
        stop=COMPLETION_REQUEST_STOPS,
        chat_options={3: {}},
    ),
)


def continuation_sources() -> list[RlDataSource[EvalDataset]]:
    return [
        RlDataSource(
            info=SourceInfo(
                id="Evalchemy:lambada_openai",
                title="lambada_openai",
                origin="Evalchemy",
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
                protocol_name="harness",
            ),
            pipeline=curate,
        )
    ]


def chat_source(
    source: RlDataSource[EvalDataset], chat_template_kwargs: dict[str, bool | str]
) -> RlDataSource[EvalDataset]:
    dataset = source.config
    assert dataset is not None and dataset.protocol.sampling is not None
    sampling = dataset.protocol.sampling.model_copy(update={"chat_template_kwargs": chat_template_kwargs})
    return replace(
        source,
        config=replace(
            dataset,
            protocol=dataset.protocol.model_copy(
                update={"operation": OperationKind.CHAT_GENERATION, "sampling": sampling}
            ),
            item_builder=partial(chat_variant_item, item_builder=dataset.item_builder),
        ),
    )


def cot_sources() -> list[RlDataSource[EvalDataset]]:
    return [
        fewshot_generation_source(
            "gsm8k_cot",
            "gsm8k_cot",
            GSM8K_SOURCE,
            1319,
            gsm8k_cot_template(),
            ("exact_match,strict-match", "exact_match,flexible-extract"),
            Sampling(
                temperature=0,
                top_p=None,
                top_k=None,
                max_tokens=None,
                stop=("Q:", "</s>", "<|im_end|>"),
                chat_template_kwargs={},
            ),
            "gsm8k_cot",
            MATH_GRADER,
        )
    ]


def _likelihood(declaration: ChoiceDeclaration, variant: ProtocolVariant) -> tuple[RlDataSource[EvalDataset], ...]:
    source = declaration.source
    shots = variant.num_fewshot
    assert shots is not None
    compiled = choice_source(
        declaration.name,
        source.repo,
        source.revision,
        source.files[0],
        declaration.population,
        HarnessTemplate(declaration.name, shots, fewshot_file=declaration.pool_file or source.files[0]),
        declaration.normalizations,
        metric_kind=declaration.metric_kind,
        scoring=declaration.scoring,
        version=declaration.fewshot_version if shots else "2026.11.02",
    )
    dataset = compiled.config
    assert dataset is not None
    return (
        replace(
            compiled,
            config=replace(dataset, protocol=dataset.protocol.model_copy(update={"benchmark": declaration.benchmark})),
        ),
    )


def _generation(declaration: GenerationDeclaration, variant: ProtocolVariant) -> tuple[RlDataSource[EvalDataset], ...]:
    shots = variant.num_fewshot
    assert shots is not None
    template = HarnessTemplate(declaration.name, shots, fewshot_file=declaration.pool_file)
    source = RlDataSource(
        info=SourceInfo(
            id="Evalchemy:" + declaration.name,
            title=declaration.name,
            origin="Evalchemy",
            family="short-answer",
            tags=("eval", "source:evalchemy", "benchmark", "family:short-answer"),
            count=declaration.population,
        ),
        config=EvalDataset(
            recipe=CurationRecipe(
                name="evalchemy-" + declaration.name,
                source=replace(declaration.source, read=partial(read_generation_file, template=template)),
                inputs=(
                    {"fewshot": replace(declaration.source, files=(declaration.pool_file,), read=None)} if shots else {}
                ),
                convert=partial(convert_generation_document, benchmark=declaration.name),
                grader=declaration.grader,
                version=scorer_revision("harness_generation"),
                intended_use=IntendedUse.EVAL,
            ),
            protocol=EvaluationProtocol(
                benchmark=declaration.name.replace("_", "-"),
                version="2026.11.02",
                operation=OperationKind.TEXT_GENERATION,
                sampling=Sampling(
                    temperature=0,
                    top_p=None,
                    top_k=None,
                    max_tokens=None,
                    stop=declaration.stop,
                    chat_template_kwargs={},
                ),
                trials=TrialPolicy(count=1, seed=None),
                num_fewshot=shots,
                metrics=tuple(
                    Metric(
                        name=metric,
                        kind=MetricKind.CONTINUOUS if metric.startswith("f1,") else MetricKind.BINARY,
                        higher_is_better=True,
                    )
                    for metric in declaration.metrics
                ),
                primary_metric=declaration.primary_metric,
                summary=SummaryKind.MEAN,
                expected_population=declaration.population,
            ),
            item_builder=generation_item,
            protocol_name="harness",
        ),
        pipeline=curate,
    )
    if variant.prompt_format is PromptFormat.CHAT:
        source = chat_source(source, declaration.chat_options[shots])
    return (source,)


def benchmarks() -> list[EvaluationBenchmark]:
    return [
        *[
            EvaluationBenchmark(
                declaration.benchmark,
                tuple(ProtocolVariant("harness", shots, None) for shots in declaration.shots),
                partial(_likelihood, declaration),
            )
            for declaration in CHOICE_PROTOCOLS
        ],
        *[
            EvaluationBenchmark(
                declaration.name.replace("_", "-"),
                tuple(
                    ProtocolVariant("harness", shots, form)
                    for shots in declaration.shots
                    for form in (
                        (PromptFormat.COMPLETION, PromptFormat.CHAT)
                        if shots in declaration.chat_options
                        else (PromptFormat.COMPLETION,)
                    )
                ),
                partial(_generation, declaration),
            )
            for declaration in GENERATION_PROTOCOLS
        ],
        *fixed_benchmarks(continuation_sources(), "harness", {"lambada_openai": "lambada"}),
        *fixed_benchmarks(cot_sources(), "cot", {"gsm8k_cot": "gsm8k"}),
    ]
