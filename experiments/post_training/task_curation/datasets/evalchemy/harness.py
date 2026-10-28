# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned native likelihood corpora with lm-eval v0.4.12 prompt semantics."""

from functools import partial

from taskcompendium.evaluation import (
    EvaluationProtocol,
    LikelihoodMetric,
    LikelihoodNormalization,
    Metric,
    MetricKind,
    Sampling,
    TrialPolicy,
)
from taskcompendium.importers.evaluation_harness import (
    HarnessTemplate,
    choice_item,
    continuation_item,
    convert_choice_document,
    convert_continuation_document,
    convert_generation_document,
    generation_item,
    original_task_name,
    read_choice_file,
    read_continuation_file,
    read_generation_file,
)
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse
from verifyit.modes.grade_harness_generation import COMPLETION_REQUEST_STOPS, GSM8K_REQUEST_STOPS

from experiments.post_training.task_curation.datasets.evalchemy.pipeline import EvalDataSet, curate
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
# Populations from cais/mmlu's pinned dataset metadata; each leaf has its own dev pool.
MMLU_POPULATIONS = {
    "abstract_algebra": 100,
    "anatomy": 135,
    "astronomy": 152,
    "business_ethics": 100,
    "clinical_knowledge": 265,
    "college_biology": 144,
    "college_chemistry": 100,
    "college_computer_science": 100,
    "college_mathematics": 100,
    "college_medicine": 173,
    "college_physics": 102,
    "computer_security": 100,
    "conceptual_physics": 235,
    "econometrics": 114,
    "electrical_engineering": 145,
    "elementary_mathematics": 378,
    "formal_logic": 126,
    "global_facts": 100,
    "high_school_biology": 310,
    "high_school_chemistry": 203,
    "high_school_computer_science": 100,
    "high_school_european_history": 165,
    "high_school_geography": 198,
    "high_school_government_and_politics": 193,
    "high_school_macroeconomics": 390,
    "high_school_mathematics": 270,
    "high_school_microeconomics": 238,
    "high_school_physics": 151,
    "high_school_psychology": 545,
    "high_school_statistics": 216,
    "high_school_us_history": 204,
    "high_school_world_history": 237,
    "human_aging": 223,
    "human_sexuality": 131,
    "international_law": 121,
    "jurisprudence": 108,
    "logical_fallacies": 163,
    "machine_learning": 112,
    "management": 103,
    "marketing": 234,
    "medical_genetics": 100,
    "miscellaneous": 783,
    "moral_disputes": 346,
    "moral_scenarios": 895,
    "nutrition": 306,
    "philosophy": 311,
    "prehistory": 324,
    "professional_accounting": 282,
    "professional_law": 1534,
    "professional_medicine": 272,
    "professional_psychology": 612,
    "public_relations": 110,
    "security_studies": 245,
    "sociology": 201,
    "us_foreign_policy": 100,
    "virology": 166,
    "world_religions": 171,
}


def choice_source(
    name: str,
    repo: str,
    revision: str,
    file: str,
    population: int,
    template: HarnessTemplate,
    normalizations: tuple[LikelihoodNormalization, ...],
) -> RlDataSource[EvalDataSet]:
    """Bind immutable evaluation rows and the separate immutable few-shot pool."""
    names = {
        LikelihoodNormalization.RAW: "acc,none",
        LikelihoodNormalization.CHARACTERS: "acc_norm,none",
        LikelihoodNormalization.BYTES: "acc_bytes,none",
    }
    inputs = (
        {"fewshot": HfSource(repo, revision, (template.fewshot_file,), SourceFormat.PARQUET)}
        if template.num_fewshot
        else {}
    )
    recipe = CurationRecipe(
        name="evalchemy-" + name,
        source=HfSource(
            repo, revision, (file,), SourceFormat.PARQUET, read=partial(read_choice_file, template=template)
        ),
        inputs=inputs,
        convert=partial(convert_choice_document, benchmark=name),
        version="1",
        intended_use=IntendedUse.EVAL,
    )
    return RlDataSource(
        info=SourceInfo(
            id="Evalchemy:" + name,
            title=name,
            origin="Evalchemy",
            family="likelihood-choice",
            tags=(
                "eval",
                "source:evalchemy",
                "single-turn",
                "benchmark",
                "family:likelihood",
                "task:" + original_task_name(template),
            ),
            count=population,
        ),
        config=EvalDataSet(
            recipe=recipe,
            protocol=EvaluationProtocol(
                benchmark=name,
                version="lm-eval-0.4.12-native-v1",
                operation="choice_likelihood",
                sampling=None,
                trials=TrialPolicy(count=1, seed=None),
                metrics=tuple(
                    Metric(
                        name=names[value],
                        kind=MetricKind.CONTINUOUS if template.name == "truthfulqa_mc2" else MetricKind.BINARY,
                        higher_is_better=True,
                    )
                    for value in normalizations
                ),
                primary_metric=names[normalizations[0]],
                reducer="mmlu_subject_weighted_mean_v1" if template.name == "mmlu" else "mean_v1",
                expected_population=population,
                likelihood_metrics=tuple(
                    LikelihoodMetric(
                        name=names[value],
                        normalization=value,
                        scoring="probability_mass" if template.name == "truthfulqa_mc2" else "most_likely",
                    )
                    for value in normalizations
                ),
            ),
            item_builder=partial(choice_item, subject=template.subject),
        ),
        pipeline=curate,
    )


def sources() -> list[RlDataSource[EvalDataSet]]:
    raw = (LikelihoodNormalization.RAW,)
    normalized = (LikelihoodNormalization.RAW, LikelihoodNormalization.CHARACTERS)
    result = [
        choice_source(
            "mmlu-" + subject,
            "cais/mmlu",
            MMLU_REVISION,
            subject + "/test-00000-of-00001.parquet",
            population,
            HarnessTemplate("mmlu", 5, subject, subject + "/dev-00000-of-00001.parquet"),
            raw,
        )
        for subject, population in MMLU_POPULATIONS.items()
    ]
    result.append(
        choice_source(
            "mmlu-smoke",
            "cais/mmlu",
            MMLU_REVISION,
            "abstract_algebra/test-00000-of-00001.parquet",
            100,
            HarnessTemplate("mmlu", 0, "abstract_algebra"),
            raw,
        )
    )
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
                title="lambada_openai",
                origin="Evalchemy",
                family="likelihood-continuation",
                tags=("eval", "source:evalchemy", "benchmark"),
                count=5153,
            ),
            config=EvalDataSet(
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
                    version="1",
                    intended_use=IntendedUse.EVAL,
                ),
                protocol=EvaluationProtocol(
                    benchmark="lambada_openai",
                    version="lm-eval-0.4.12-native-v1",
                    operation="continuation_likelihood",
                    sampling=None,
                    trials=TrialPolicy(count=1, seed=None),
                    metrics=(
                        Metric(name="acc,none", kind=MetricKind.BINARY, higher_is_better=True),
                        Metric(name="perplexity,none", kind=MetricKind.CONTINUOUS, higher_is_better=False),
                    ),
                    primary_metric="acc,none",
                    reducer="lambada_perplexity_from_mean_log_likelihood_v1",
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


def generation_sources() -> list[RlDataSource[EvalDataSet]]:
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
                    title=name,
                    origin="Evalchemy",
                    family="short-answer",
                    tags=("eval", "source:evalchemy", "benchmark", "family:short-answer", "task:" + profile),
                    count=population,
                ),
                config=EvalDataSet(
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
                        version="1",
                        intended_use=IntendedUse.EVAL,
                    ),
                    protocol=EvaluationProtocol(
                        benchmark=name,
                        version="lm-eval-0.4.12-vendor-native-v1",
                        operation="text_generation",
                        sampling=Sampling(
                            temperature=0,
                            top_p=1,
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
                        reducer="mean_v1",
                        expected_population=population,
                    ),
                    item_builder=generation_item,
                ),
                pipeline=curate,
            )
        )
    return result
