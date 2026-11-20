# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ChemBench zero-shot chat generation with the authors' instruction prompts and answer parsing.

ChemBench (Mirza et al., Nature Chemistry 2025) asks 2,785 chemistry questions across nine topic files:
2,542 multiple choice, 395 of them with several correct options, and 243 with a numeric answer. Scores
are fraction correct overall and per topic. The dataset and the reference runner, lamalab-org/chembench,
are MIT-licensed.

Decoding is greedy, as in the authors' published runs, and thinking stays enabled for models that support
it. The authors' runner falls back to an LLM extractor for untagged answers; this protocol scores them zero.
"""

from taskcompendium.evaluation.spec import (
    EvaluationProtocol,
    Metrics,
    OperationKind,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.chembench import chembench_item, convert_chembench
from taskcompendium.importers.evalchemy.grader import scorer_revision
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.environments import BASIC_GRADER
from experiments.evaluation.prompt_compilation import evaluation_source
from experiments.post_training.task_curation.pipeline import HfSource

CHEMBENCH_REVISION = "6e1d25748952393f44e35b8e85bbe567246a6430"
CHEMBENCH_TOPICS = (
    "analytical_chemistry",
    "chemical_preference",
    "general_chemistry",
    "inorganic_chemistry",
    "materials_science",
    "organic_chemistry",
    "physical_chemistry",
    "technical_chemistry",
    "toxicity_and_safety",
)
CHEMBENCH_FILES = tuple(f"{topic}/train-00000-of-00001.parquet" for topic in CHEMBENCH_TOPICS)
CHEMBENCH_POPULATION = 2785
VERSION = "2026.11.19"


def benchmarks() -> list[EvaluationBenchmark]:
    protocol = EvaluationProtocol(
        benchmark="chembench",
        version=VERSION,
        operation=OperationKind.CHAT_GENERATION,
        sampling=Sampling(
            temperature=0.0,
            top_p=None,
            top_k=None,
            max_tokens=None,
            stop=(),
            chat_template_kwargs={"enable_thinking": True},
        ),
        trials=TrialPolicy(count=1, seed=None),
        metrics=(Metrics.ACCURACY,),
        primary_metric="accuracy",
        summary=SummaryKind.MEAN,
        expected_population=CHEMBENCH_POPULATION,
    )
    source = evaluation_source(
        name="chembench",
        source=HfSource("jablonkagroup/ChemBench", CHEMBENCH_REVISION, CHEMBENCH_FILES, SourceFormat.PARQUET),
        convert=convert_chembench,
        version=VERSION + "-" + scorer_revision("chembench"),
        protocol=protocol,
        item_builder=chembench_item,
        grader=BASIC_GRADER,
    )
    return fixed_benchmarks([source], "chat")
