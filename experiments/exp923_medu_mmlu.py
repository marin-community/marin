import os
from dataclasses import dataclass

from experiments.datashop.datashop_datasets import datashop_dclm_annotation_subset, datashop_dclm_pretraining_subset
from experiments.datashop.datashop_runner import DatashopRunner, DatashopRunnerConfig
from experiments.eval_datasets import mmlu_subject_eval
from marin.datashop.pipeline import CorpusContent
from marin.execution.executor import executor_main

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


@dataclass
class MeduMMLUConfig:
    subset_names: list[str]
    experiment_name: str
    annotator_model_name: str = "Llama-3.3-70B-Instruct"
    pretraining_data_path: str = datashop_dclm_pretraining_subset
    annotator_data_path: str = datashop_dclm_annotation_subset


class MMLUMeduPipeline(DatashopRunner):
    def __init__(self, config: MeduMMLUConfig):

        self.config = config
        self.corpus_contents = self._create_corpus_contents()
        super().__init__(
            DatashopRunnerConfig(
                experiment_name=self.config.experiment_name,
                annotator_model_name=self.config.annotator_model_name,
                pretraining_data_path=self.config.pretraining_data_path,
                annotator_data_path=self.config.annotator_data_path,
                corpus_content_paths=self.corpus_contents,
            )
        )

    def _create_corpus_contents(self):
        # Download the MMLU dataset
        corpus_contents = []
        for subject in self.config.subset_names:
            filepath = os.path.join(
                os.getenv("MARIN_PREFIX"), mmlu_subject_eval.name, "cais", f"mmlu-{subject}-dev-evaluation.jsonl.gz"
            )
            corpus_contents.append(CorpusContent(content=filepath, content_type="filepath", prompt_column="prompt"))

        return corpus_contents


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
    executor_main(
        mmlu_science_pipeline.get_all_steps()
        + mmlu_engineering_pipeline.get_all_steps()
        + mmlu_social_sciences_pipeline.get_all_steps()
        + mmlu_humanities_pipeline.get_all_steps()
        + mmlu_other_pipeline.get_all_steps()
    )
