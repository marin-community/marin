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
# MMLU's pinned subjects from cais/mmlu; each leaf has its own dev pool.
MMLU_SUBJECTS = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
)


def sources(shots: int = 5) -> list[RlDataSource[EvaluationCorpusRecipe]]:
    raw = (LikelihoodNormalization.RAW,)
    result = []
    for subject in MMLU_SUBJECTS:
        result.append(
            lm_eval_multiple_choice(
                name="mmlu-" + subject,
                benchmark="mmlu",
                source=HfSource(
                    "cais/mmlu", MMLU_REVISION, (subject + "/test-00000-of-00001.parquet",), SourceFormat.PARQUET
                ),
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
