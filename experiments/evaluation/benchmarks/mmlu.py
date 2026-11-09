# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Classic MMLU subject populations and pinned development pools."""

from taskcompendium.evaluation.spec import LikelihoodNormalization
from taskcompendium.importers.evaluation_harness import HarnessTemplate
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.evaluation.benchmark import EvaluationBenchmark, ProtocolVariant
from experiments.evaluation.preparation import EvaluationCorpusRecipe
from experiments.evaluation.prompt_compilation import lm_eval_multiple_choice
from experiments.post_training.task_curation.pipeline import HfSource
from experiments.post_training.task_curation.source import RlDataSource

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


def sources(shots: int = 5) -> list[RlDataSource[EvaluationCorpusRecipe]]:
    raw = (LikelihoodNormalization.RAW,)
    result = []
    for subject, population in MMLU_POPULATIONS.items():
        result.append(
            lm_eval_multiple_choice(
                name="mmlu-" + subject,
                benchmark="mmlu",
                source=HfSource(
                    "cais/mmlu", MMLU_REVISION, (subject + "/test-00000-of-00001.parquet",), SourceFormat.PARQUET
                ),
                population=population,
                template=HarnessTemplate(
                    name="mmlu",
                    num_fewshot=shots,
                    subject=subject,
                    fewshot_file=subject + "/dev-00000-of-00001.parquet",
                ),
                normalizations=raw,
                version="2026.11.02",
                partition=subject,
            )
        )
    return result


def build(variant: ProtocolVariant) -> tuple[RlDataSource[EvaluationCorpusRecipe], ...]:
    assert variant.num_fewshot is not None
    return tuple(sources(variant.num_fewshot))


def benchmarks() -> list[EvaluationBenchmark]:
    return [EvaluationBenchmark("mmlu", tuple(ProtocolVariant("harness", shots, None) for shots in (5, 0)), build)]
