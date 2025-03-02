import os
from dataclasses import dataclass

from experiments.medu.defaults import default_label
from experiments.medu.medu_datasets import medu_dclm_annotation_subset
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.generation.medu import CorpusContent
from operations.download.huggingface.download import DownloadConfig
from operations.download.huggingface.download_hf import download_hf
from operations.raw2json.huggingface.qa.raw2json import DatasetConversionConfig, OutputFormatOptions, raw2json


@dataclass
class MeduMMLUConfig:
    subset_names: list[str]
    experiment_name: str


# TODO(chris): Use output_path_of the mmlu eval set
MMLU_BASE_PATH = (
    "gs://marin-us-central2/evaluation/mmlu-eval-subject-fc0515/cais/mmlu-{subject_name}-dev-evaluation.jsonl.gz"
)

mmlu_raw = ExecutorStep(
    name="raw/cais/mmlu",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="cais/mmlu",
        revision=versioned("c30699e"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.md"],
    ),
    override_output_path="raw/cais/mmluhf",
)

mmlu_subject_eval = ExecutorStep(
    name="evaluation/mmlu-eval-subject",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="cais/mmlu",
        subsets=["*"],
        splits=["dev", "validation"],
        input_path=mmlu_raw,
        hf_path="cais/mmlu",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("evaluation"),
        prompt_key="question",
        options_key="choices",
        answer_idx_key="answer",
        answer_labels=["A", "B", "C", "D"],
        exclude_subsets=["all", "auxiliary_train"],
    ),
    override_output_path="evaluation/mmlu-eval-subject",
)

# Humanities
humanities = [
    "philosophy",
    "world_religions",
    "high_school_european_history",
    "high_school_us_history",
    "high_school_world_history",
    "prehistory",
    "moral_disputes",
    "moral_scenarios",
]

# Mathematics
mathematics = [
    "abstract_algebra",
    "college_mathematics",
    "elementary_mathematics",
    "high_school_mathematics",
    "high_school_statistics",
]

# Science
science = [
    "anatomy",
    "astronomy",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_physics",
    "conceptual_physics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_physics",
    "medical_genetics",
    "nutrition",
    "virology",
    "human_aging",
]

# Engineering
engineering = [
    "electrical_engineering",
    "computer_security",
    "college_computer_science",
    "high_school_computer_science",
    "machine_learning",
]

# Social Sciences
social_sciences = [
    "business_ethics",
    "econometrics",
    "global_facts",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "high_school_psychology",
    "sociology",
    "us_foreign_policy",
    "public_relations",
    "security_studies",
]

# Other
other = [
    "formal_logic",
    "logical_fallacies",
    "international_law",
    "jurisprudence",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "management",
    "marketing",
    "miscellaneous",
]


def default_mmlu_labeling(config: MeduMMLUConfig):
    mmlu_base_path = mmlu_subject_eval.name
    corpus_contents = []
    for subject in config.subset_names:
        filepath = os.path.join(os.getenv("MARIN_PREFIX"), mmlu_base_path, f"mmlu-{subject}-dev-evaluation.jsonl.gz")
        corpus_contents.append(CorpusContent(content=filepath, content_type="filepath"))

    return default_label(
        documents_to_be_labeled=medu_dclm_annotation_subset,
        targeted_documents=corpus_contents,
        experiment_name=config.experiment_name,
    )


mmlu_humanities_labeled = default_mmlu_labeling(
    MeduMMLUConfig(subset_names=humanities, experiment_name="mmlu-humanities")
)
mmlu_mathematics_labeled = default_mmlu_labeling(
    MeduMMLUConfig(subset_names=mathematics, experiment_name="mmlu-mathematics")
)
mmlu_science_labeled = default_mmlu_labeling(MeduMMLUConfig(subset_names=science, experiment_name="mmlu-science"))
mmlu_engineering_labeled = default_mmlu_labeling(
    MeduMMLUConfig(subset_names=engineering, experiment_name="mmlu-engineering")
)
mmlu_social_sciences_labeled = default_mmlu_labeling(
    MeduMMLUConfig(subset_names=social_sciences, experiment_name="mmlu-social-sciences")
)
mmlu_other_labeled = default_mmlu_labeling(MeduMMLUConfig(subset_names=other, experiment_name="mmlu-other"))


if __name__ == "__main__":
    executor_main([mmlu_mathematics_labeled])
