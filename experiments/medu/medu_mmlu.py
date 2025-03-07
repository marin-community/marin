import os
from dataclasses import dataclass

from experiments.medu.defaults import default_label, default_quality_filter_and_consolidate, default_quality_filter_model
from experiments.medu.medu_datasets import medu_dclm_pretraining_subset
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.generation.medu import CorpusContent
from operations.download.huggingface.download import DownloadConfig
from operations.download.huggingface.download_hf import download_hf
from operations.raw2json.huggingface.qa.raw2json import DatasetConversionConfig, OutputFormatOptions, raw2json


@dataclass
class MeduMMLUConfig:
    subset_names: list[str]
    experiment_name: str


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
    "formal_logic",
    "logical_fallacies",
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
    "professional_medicine",
    "professional_psychology",
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
# Law, Economics, Geography, Government, Politics, Psychology, Sociology, International Law
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
    "management",
    "marketing",
    "professional_accounting",
    "jurisprudence",
    "professional_law",
    "international_law",
]

# Other
other = [
    "miscellaneous",
]


class MMLUMeduPipeline:
    def __init__(self, config: MeduMMLUConfig):
        self.config = config
        self.mmlu_base_path = mmlu_subject_eval.name
        self.corpus_contents = self._create_corpus_contents()
        self.pretraining_data_path = medu_dclm_pretraining_subset
        self.pretraining_data_name = "medu-dclm-pretraining-subset"

        # To be populated by the default_mmlu_labeling step
        self.labeled_documents = self.default_mmlu_labeling()
        self.encoder_model = default_quality_filter_model(self.labeled_documents, self.config.experiment_name)
        self.filtered_documents = default_quality_filter_and_consolidate(
            self.encoder_model, self.pretraining_data_path, self.pretraining_data_name, self.config.experiment_name
        )

    def _create_corpus_contents(self):
        corpus_contents = []
        for subject in self.config.subset_names:
            filepath = os.path.join(
                os.getenv("MARIN_PREFIX"), self.mmlu_base_path, "cais", f"mmlu-{subject}-dev-evaluation.jsonl.gz"
            )
            corpus_contents.append(CorpusContent(content=filepath, content_type="filepath", prompt_column="prompt"))

        return corpus_contents

    # TODO(chris): Turn into a step so we can use output path of MMLU step
    def default_mmlu_labeling(self):
        return default_label(
            documents_to_be_labeled="gs://marin-us-east5/documents/medu-datasets/medu-dclm-annotation-subset-e12303/medu-dclm-annotation-subset-e12303",
            # TODO(chris): Use direct path for now since we don't have DCLM downloaded on east5.
            # documents_to_be_labeled=medu_dclm_annotation_subset,
            targeted_documents=self.corpus_contents,
            experiment_name=self.config.experiment_name,
        )

    def get_executor_steps(self):
        return [self.labeled_documents, self.encoder_model]

    def run(self):
        executor_main(self.get_executor_steps())


# mmlu_humanities_labeled = default_mmlu_labeling(
#     MeduMMLUConfig(subset_names=humanities, experiment_name="mmlu-humanities")
# )
# mmlu_mathematics_labeled = default_mmlu_labeling(
#     MeduMMLUConfig(subset_names=mathematics, experiment_name="mmlu-mathematics")
# )
# mmlu_science_labeled = default_mmlu_labeling(MeduMMLUConfig(subset_names=science, experiment_name="mmlu-science"))
# mmlu_engineering_labeled = default_mmlu_labeling(
#     MeduMMLUConfig(subset_names=engineering, experiment_name="mmlu-engineering")
# )
# mmlu_social_sciences_labeled = default_mmlu_labeling(
#     MeduMMLUConfig(subset_names=social_sciences, experiment_name="mmlu-social-sciences")
# )
# mmlu_other_labeled = default_mmlu_labeling(MeduMMLUConfig(subset_names=other, experiment_name="mmlu-other"))

# mmlu_mathematics_dataset = default_dataset_creation(mmlu_mathematics_labeled, "mmlu-mathematics")
# mmlu_mathematics_model = default_encoder_model(mmlu_mathematics_dataset, "mmlu-mathematics")

mmlu_science_pipeline = MMLUMeduPipeline(MeduMMLUConfig(subset_names=science, experiment_name="mmlu-science"))
mmlu_engineering_pipeline = MMLUMeduPipeline(
    MeduMMLUConfig(subset_names=engineering, experiment_name="mmlu-engineering")
)
mmlu_social_sciences_pipeline = MMLUMeduPipeline(
    MeduMMLUConfig(subset_names=social_sciences, experiment_name="mmlu-social-sciences")
)
mmlu_humanities_pipeline = MMLUMeduPipeline(MeduMMLUConfig(subset_names=humanities, experiment_name="mmlu-humanities"))
mmlu_other_pipeline = MMLUMeduPipeline(MeduMMLUConfig(subset_names=other, experiment_name="mmlu-other"))

if __name__ == "__main__":
    mmlu_science_pipeline.run()
