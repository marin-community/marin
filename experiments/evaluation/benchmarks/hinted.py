# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hinted GSM8K and MATH-500 base-model completion protocols (Hinted Scaling Laws, arXiv 2611.02270).

Each task has an unhinted H = 0 protocol and one protocol per hint method covering
H = 0.1, ..., 1.0. Every hinted item carries ``hint_method`` and ``hint_level`` groups, so the
reducer reports one score per level, such as ``hint_level=0.3/accuracy``. Every row keeps its
``reference_id`` and reference solution in the task evidence. Sampling, stops, output budgets and
trial counts follow the paper's Delphi runs (tatsu-lab/hinted-scaling-laws
``experiments/query_hinted_benchmarks``): temperature 0.4, top-p 1, top-k 1000 for hinted runs and
disabled for H = 0, 512 output tokens, 64 trials at H = 0 and 32 at H > 0 on GSM8K. MATH-500,
which the paper does not evaluate, uses the same sampling with a 1024-token output budget, 32
trials at H = 0 and 16 at H > 0.

References derive programmatically from the pinned sources. GSM8K (MIT) uses the first 256 test
problems; MATH-500 (MIT) uses all 500 problems in the Minerva four-shot format.
"""

from dataclasses import replace
from functools import partial

from taskcompendium.evaluation.spec import (
    EvaluationProtocol,
    Metric,
    MetricKind,
    OperationKind,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.grader import scorer_revision
from taskcompendium.importers.evalchemy.hinted import (
    GSM8K_HINTED_POPULATION,
    GSM8K_SHOTS,
    HINT_LEVELS,
    MINERVA_SHOTS,
    HintMethod,
    convert_hinted,
    hinted_item,
    read_hinted_gsm8k,
    read_hinted_math,
)
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.environments import BASIC_GRADER, MATH_GRADER
from experiments.evaluation.benchmarks.gsm8k import GSM8K_SOURCE
from experiments.evaluation.benchmarks.math import MATH500_REVISION
from experiments.evaluation.preparation import EvaluationCorpusRecipe
from experiments.evaluation.prompt_compilation import evaluation_source
from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.pipeline import HfSource, RecipeSource
from experiments.post_training.task_curation.source import RlDataSource

VERSION = "2026.11.20"
PROTOCOL = "hinted"
GSM8K_POOL_FILE = "main/train-00000-of-00001.parquet"
MATH500_SOURCE = HfSource("HuggingFaceH4/MATH-500", MATH500_REVISION, ("test.jsonl",), SourceFormat.JSONL)
MATH500_POPULATION = 500
GSM8K_STOPS = ("Question:", "</s>", "<|im_end|>")
MINERVA_STOPS = ("Problem:",)
GSM8K_MAX_TOKENS = 512
# The longest H=1 MATH-500 prompt is about 2,300 tokens, so this budget fits the 4,096-token context of the
# Delphi and hero-ladder checkpoints these evals target.
MATH_MAX_TOKENS = 1024
HINTED_TEMPERATURE = 0.4
HINTED_TOP_K = 1000
# vLLM reads -1 as "no top-k"; the paper's unhinted runs disable top-k.
UNHINTED_TOP_K = -1
GSM8K_TRIALS = {HintMethod.NONE: 64, HintMethod.TRUNCATION: 32, HintMethod.MASKING: 32}
MATH500_TRIALS = {HintMethod.NONE: 32, HintMethod.TRUNCATION: 16, HintMethod.MASKING: 16}
GSM8K_METRICS = (Metric(name="accuracy", kind=MetricKind.BINARY, higher_is_better=True),)
MINERVA_METRICS = tuple(
    Metric(name=name, kind=MetricKind.BINARY, higher_is_better=True) for name in ("exact_match,none", "math_verify,none")
)


def hinted_protocol(
    *,
    benchmark: str,
    method: HintMethod,
    trials: int,
    metrics: tuple[Metric, ...],
    shots: int,
    problems: int,
    stop: tuple[str, ...],
    max_tokens: int,
) -> EvaluationProtocol:
    return EvaluationProtocol(
        benchmark=benchmark,
        version=VERSION,
        operation=OperationKind.TEXT_GENERATION,
        sampling=Sampling(
            temperature=HINTED_TEMPERATURE,
            top_p=1.0,
            top_k=UNHINTED_TOP_K if method is HintMethod.NONE else HINTED_TOP_K,
            max_tokens=max_tokens,
            stop=stop,
            chat_template_kwargs={},
        ),
        trials=TrialPolicy(count=trials, seed=None),
        metrics=metrics,
        primary_metric=metrics[0].name,
        num_fewshot=shots,
        summary=SummaryKind.MEAN,
        expected_population=problems if method is HintMethod.NONE else problems * len(HINT_LEVELS),
    )


def hinted_source(
    *,
    name: str,
    source: RecipeSource,
    inputs: dict[str, RecipeSource],
    scorer: str,
    grader: Environment,
    protocol: EvaluationProtocol,
) -> RlDataSource[EvaluationCorpusRecipe]:
    built = evaluation_source(
        name=name,
        source=source,
        inputs=inputs,
        convert=partial(convert_hinted, scorer=scorer, benchmark=protocol.benchmark),
        version=VERSION + "-" + scorer_revision(scorer),
        protocol=protocol,
        item_builder=hinted_item,
        grader=grader,
    )
    assert built.config is not None
    return replace(built, config=replace(built.config, protocol_name=PROTOCOL))


def gsm8k_sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    return [
        hinted_source(
            name=f"gsm8k-hinted-{method.value}",
            source=replace(
                GSM8K_SOURCE,
                read=partial(
                    read_hinted_gsm8k,
                    method=method,
                    pool_file=GSM8K_POOL_FILE,
                    source_row_prefix=f"{GSM8K_SOURCE.repo}@{GSM8K_SOURCE.revision[:8]}:test",
                ),
            ),
            inputs={"fewshot": replace(GSM8K_SOURCE, files=(GSM8K_POOL_FILE,))},
            scorer="gsm8k_first_hash",
            grader=BASIC_GRADER,
            protocol=hinted_protocol(
                benchmark=f"gsm8k-hinted-{method.value}",
                method=method,
                trials=trials,
                metrics=GSM8K_METRICS,
                shots=GSM8K_SHOTS,
                problems=GSM8K_HINTED_POPULATION,
                stop=GSM8K_STOPS,
                max_tokens=GSM8K_MAX_TOKENS,
            ),
        )
        for method, trials in GSM8K_TRIALS.items()
    ]


def math500_sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    return [
        hinted_source(
            name=f"math500-hinted-{method.value}",
            source=replace(
                MATH500_SOURCE,
                read=partial(
                    read_hinted_math,
                    method=method,
                    source_row_prefix=f"{MATH500_SOURCE.repo}@{MATH500_SOURCE.revision[:8]}",
                ),
            ),
            inputs={},
            scorer="minerva_math",
            grader=MATH_GRADER,
            protocol=hinted_protocol(
                benchmark=f"math500-hinted-{method.value}",
                method=method,
                trials=trials,
                metrics=MINERVA_METRICS,
                shots=MINERVA_SHOTS,
                problems=MATH500_POPULATION,
                stop=MINERVA_STOPS,
                max_tokens=MATH_MAX_TOKENS,
            ),
        )
        for method, trials in MATH500_TRIALS.items()
    ]


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks([*gsm8k_sources(), *math500_sources()])
