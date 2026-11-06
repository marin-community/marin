# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Classic MMLU subject populations and pinned development pools."""

from dataclasses import replace

from taskcompendium.evaluation.spec import LikelihoodNormalization
from taskcompendium.importers.evaluation_harness import HarnessTemplate

from experiments.post_training.task_curation.datasets.evalchemy.protocols import choice_source
from experiments.post_training.task_curation.evaluation import EvalDataset
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


def sources() -> list[RlDataSource[EvalDataset]]:
    raw = (LikelihoodNormalization.RAW,)
    result = []
    for shots in (0, 5):
        for subject, population in MMLU_POPULATIONS.items():
            source = choice_source(
                "mmlu-" + subject if shots == 5 else "mmlu-0shot-" + subject,
                "cais/mmlu",
                MMLU_REVISION,
                subject + "/test-00000-of-00001.parquet",
                population,
                HarnessTemplate("mmlu", shots, subject, subject + "/dev-00000-of-00001.parquet"),
                raw,
            )
            dataset = source.config
            assert dataset is not None
            result.append(
                replace(
                    source,
                    config=replace(
                        dataset,
                        protocol=dataset.protocol.model_copy(
                            update={
                                "benchmark": "mmlu" if shots == 5 else "mmlu-0shot",
                            }
                        ),
                    ),
                )
            )
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
    return result
