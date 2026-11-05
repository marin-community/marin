# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact additional protocols used by the checkpoint evaluation menus."""

from dataclasses import replace
from functools import partial

from taskcompendium.evaluation.spec import (
    EvaluationProtocol,
    LikelihoodMetric,
    Metric,
    OperationKind,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.identity import canonical_sha256
from taskcompendium.importers.checkpoint_harness import (
    CheckpointTemplate,
    checkpoint_prompts,
    convert_checkpoint_generation,
    read_checkpoint_file,
    read_humaneval_fewshot,
)
from taskcompendium.importers.evalchemy.grader import scorer_revision
from taskcompendium.importers.evaluation_harness import (
    HarnessTemplate,
    choice_item,
    convert_choice_document,
    generation_item,
    read_generation_file,
)
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.evalchemy.code import sources as python_sources
from experiments.post_training.task_curation.datasets.evalchemy.environments import MATH_GRADER
from experiments.post_training.task_curation.datasets.evalchemy.harness import generation_sources
from experiments.post_training.task_curation.datasets.evalchemy.reasoning import (
    GPQA_REVISION,
    MMLU_PRO_REVISION,
    MMLU_TEST_FILE,
)
from experiments.post_training.task_curation.evaluation import EvalDataset, curate
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

BBH_REVISION = "b5306be6f827cfafbb545ff5a51f96916029b0fd"
MATH_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"


def checkpoint_source(
    name: str,
    benchmark: str,
    source: HfSource,
    population: int,
    template: CheckpointTemplate,
    metrics: tuple[str, ...],
    operation: OperationKind,
) -> RlDataSource[EvalDataset]:
    generation = operation is OperationKind.TEXT_GENERATION
    metric_defs = tuple(Metric(name=metric, kind="binary", higher_is_better=True) for metric in metrics)
    stops = {
        "gsm8k_cot": ("Q:", "</s>", "<|im_end|>"),
        "bbh_cot_fewshot": ("</s>", "Q", "\n\n"),
        "minerva_math": ("Problem:",),
    }
    protocol = EvaluationProtocol(
        benchmark=benchmark,
        version="2026.11.05",
        operation=operation,
        sampling=(
            Sampling(
                temperature=0,
                top_p=None,
                top_k=None,
                max_tokens=1024 if template.family == "bbh_cot_fewshot" else None,
                stop=stops[template.family],
                chat_template_kwargs={},
            )
            if generation
            else None
        ),
        trials=TrialPolicy(count=1, seed=None),
        metrics=metric_defs,
        primary_metric=metrics[-1] if not generation else metrics[0],
        num_fewshot=template.shots,
        summary=SummaryKind.MEAN,
        expected_population=population,
        likelihood_metrics=(
            ()
            if generation
            else tuple(
                LikelihoodMetric(name=metric, normalization="characters" if "acc_norm" in metric else "raw")
                for metric in metrics
            )
        ),
    )
    return RlDataSource(
        info=SourceInfo(
            id="Checkpoint:" + name,
            title=name,
            origin="lm-eval",
            family="checkpoint",
            tags=("eval", "benchmark"),
            count=population,
        ),
        config=EvalDataset(
            CurationRecipe(
                name="checkpoint-" + name,
                source=replace(source, read=partial(read_checkpoint_file, template=template)),
                inputs=(
                    {"fewshot": HfSource(source.repo, source.revision, (template.fewshot_file,), SourceFormat.PARQUET)}
                    if template.fewshot_file
                    else {}
                ),
                convert=(
                    convert_checkpoint_generation
                    if generation
                    else partial(convert_choice_document, benchmark=benchmark)
                ),
                version=canonical_sha256(checkpoint_prompts())[:16]
                + (scorer_revision("checkpoint_generation") if generation else ""),
                intended_use=IntendedUse.EVAL,
                grader=MATH_GRADER if generation else None,
            ),
            protocol,
            generation_item if generation else partial(choice_item, subject=template.subject),
        ),
        pipeline=curate,
    )


def sources() -> list[RlDataSource[EvalDataset]]:
    likelihood = OperationKind.CHOICE_LIKELIHOOD
    generation = OperationKind.TEXT_GENERATION
    result = []
    for name, repo, pin, file, population, shots, pool, metrics in (
        (
            "commonsense_qa",
            "tau/commonsense_qa",
            "94630fe30dad47192a8546eb75f094926d47e155",
            "data/validation-00000-of-00001.parquet",
            1221,
            10,
            "data/train-00000-of-00001.parquet",
            ("acc,none",),
        ),
        (
            "copa",
            "aps/super_glue",
            "3de24cf8022e94f4ee4b9d55a6f539891524d646",
            "copa/validation-00000-of-00001.parquet",
            100,
            0,
            "",
            ("acc,none",),
        ),
        (
            "wsc273",
            "lighteval/winograd_wsc",
            "5df8523abfe0c256d133b9887328e3e5697cb67a",
            "wsc273/test-00000-of-00001.parquet",
            273,
            0,
            "",
            ("acc,none",),
        ),
        (
            "agieval_lsat_ar",
            "hails/agieval-lsat-ar",
            "052cc636b612f5563329dd182fb6c2cad56681c8",
            "data/test-00000-of-00001.parquet",
            230,
            3,
            "",
            ("acc,none", "acc_norm,none"),
        ),
        (
            "leaderboard_mmlu_pro",
            "TIGER-Lab/MMLU-Pro",
            MMLU_PRO_REVISION,
            MMLU_TEST_FILE,
            12032,
            5,
            "data/validation-00000-of-00001.parquet",
            ("acc,none",),
        ),
    ):
        result.append(
            checkpoint_source(
                name,
                name,
                HfSource(repo, pin, (file,), SourceFormat.PARQUET),
                population,
                CheckpointTemplate(name, shots=shots, fewshot_file=pool),
                metrics,
                likelihood,
            )
        )
    for subject, population in (("diamond", 198), ("main", 448), ("extended", 546)):
        result.append(
            checkpoint_source(
                "leaderboard_gpqa-" + subject,
                "leaderboard_gpqa",
                HfSource("Idavidrein/gpqa", GPQA_REVISION, ("gpqa_" + subject + ".csv",), SourceFormat.CSV),
                population,
                CheckpointTemplate("leaderboard_gpqa", subject),
                ("acc_norm,none",),
                likelihood,
            )
        )
    for key in checkpoint_prompts():
        if not key.startswith(("bbh/", "bbh_cot/")):
            continue
        family, subject = key.split("/", 1)
        population = {"causal_judgement": 187, "penguins_in_a_table": 146, "snarks": 178}.get(subject, 250)
        benchmark = "leaderboard_bbh" if family == "bbh" else "bbh_cot_fewshot"
        result.append(
            checkpoint_source(
                benchmark + "-" + subject,
                benchmark,
                HfSource(
                    "SaylorTwift/bbh", BBH_REVISION, (subject + "/test-00000-of-00001.parquet",), SourceFormat.PARQUET
                ),
                population,
                CheckpointTemplate(benchmark, subject, shots=3),
                ("acc_norm,none",) if family == "bbh" else ("exact_match,get-answer",),
                likelihood if family == "bbh" else generation,
            )
        )
    result.append(
        checkpoint_source(
            "gsm8k_cot",
            "gsm8k_cot",
            HfSource(
                "openai/gsm8k",
                "740312add88f781978c0658806c59bc2815b9866",
                ("main/test-00000-of-00001.parquet",),
                SourceFormat.PARQUET,
            ),
            1319,
            CheckpointTemplate("gsm8k_cot", shots=8),
            ("exact_match,strict-match", "exact_match,flexible-extract"),
            generation,
        )
    )
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
            checkpoint_source(
                "minerva_math-" + subject,
                "minerva_math",
                HfSource(
                    "EleutherAI/hendrycks_math",
                    MATH_REVISION,
                    (subject + "/test-00000-of-00001.parquet",),
                    SourceFormat.PARQUET,
                ),
                population,
                CheckpointTemplate("minerva_math", subject, shots=4),
                ("exact_match,none", "math_verify,none"),
                generation,
            )
        )
    for source in generation_sources():
        dataset = source.config
        assert dataset is not None
        profile = dataset.recipe.name.removeprefix("evalchemy-")
        if profile not in {"nq_open", "triviaqa", "drop"}:
            continue
        name = profile + "-0shot"
        assert isinstance(dataset.recipe.source, HfSource)
        result.append(
            replace(
                source,
                info=replace(source.info, id="Checkpoint:" + name, title=name),
                config=replace(
                    dataset,
                    recipe=replace(
                        dataset.recipe,
                        name="checkpoint-" + name,
                        inputs={},
                        source=replace(
                            dataset.recipe.source,
                            read=partial(read_generation_file, template=HarnessTemplate(profile, 0)),
                        ),
                    ),
                    protocol=dataset.protocol.model_copy(update={"benchmark": name, "num_fewshot": 0}),
                ),
            )
        )
    human = next(source for source in python_sources() if source.name == "evalchemy-humaneval")
    dataset = human.config
    assert dataset is not None
    assert isinstance(dataset.recipe.source, HfSource)
    result.append(
        replace(
            human,
            info=replace(human.info, id="Checkpoint:humaneval-10shot", title="humaneval-10shot"),
            config=replace(
                dataset,
                recipe=replace(
                    dataset.recipe,
                    name="checkpoint-humaneval-10shot",
                    source=replace(dataset.recipe.source, read=read_humaneval_fewshot),
                ),
                protocol=dataset.protocol.model_copy(update={"benchmark": "humaneval-10shot", "num_fewshot": 10}),
            ),
        )
    )
    return result
